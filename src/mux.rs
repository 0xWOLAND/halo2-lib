use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::Field,
    circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value},
    halo2curves::pasta::pallas,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, Selector},
    poly::Rotation,
};

type Element<F> = AssignedCell<F, F>;

#[derive(Clone, Debug)]
pub struct MuxConfig {
    advice: [Column<Advice>; 3],
    sel: Column<Fixed>,
    s: Selector,
}

#[derive(Clone, Debug)]
pub struct MuxChip {
    config: MuxConfig,
    _marker: PhantomData<pallas::Base>,
}

impl Chip<pallas::Base> for MuxChip {
    type Config = MuxConfig;

    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl MuxChip {
    pub fn new(config: MuxConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure<F: Field>(meta: &mut ConstraintSystem<F>) -> MuxConfig {
        let advice = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let s = meta.selector();
        let sel = meta.fixed_column();

        meta.create_gate("mux", |meta| {
            let s = meta.query_selector(s);

            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[2], Rotation::cur());
            let sel = meta.query_fixed(sel, Rotation::cur());

            vec![s * (((Expression::Constant(F::ONE) - sel.clone()) * a + sel * b) - out)]
        });

        MuxConfig {
            advice: advice.try_into().unwrap(),
            sel,
            s,
        }
    }

    pub fn mux<F: Field>(
        &self,
        mut layouter: impl Layouter<F>,
        a: Value<F>,
        b: Value<F>,
        sel: Value<F>,
        row: usize,
    ) -> Result<Element<F>, Error> {
        layouter.assign_region(
            || "sel",
            |mut region| {
                self.config.s.enable(&mut region, row)?;
                let out = a * (Value::known(F::ONE) - sel) + b * sel;

                let _sel = region.assign_fixed(|| "sel", self.config.sel, row, || sel)?;
                let _in_a = region.assign_advice(|| "in_a", self.config.advice[0], row, || a)?;
                let _in_b = region.assign_advice(|| "in_b", self.config.advice[1], row, || b)?;
                let out = region.assign_advice(|| "out", self.config.advice[2], row, || out)?;

                Ok(out)
            },
        )
    }
}

#[derive(Clone, Default)]
struct MuxCircuit<'a, F: Field, const L: usize> {
    pub a: &'a [Value<F>],
    pub b: &'a [Value<F>],
    pub mux: Value<F>,
}

impl<F: Field, const L: usize> Circuit<F> for MuxCircuit<'_, F, L> {
    type Config = MuxConfig;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        MuxChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = MuxChip::new(config);
        for i in 0..L {
            let _ = chip.mux(
                layouter.namespace(|| format!("mux_{}", i)),
                self.a[i],
                self.b[i],
                self.mux,
                i,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use halo2_proofs::{dev::MockProver, halo2curves::pasta::Fp};

    use super::*;

    #[test]
    fn test() {
        const LEN: usize = 8;
        let a = [1, 2, 3, 4, 5, 6, 7, 8].map(|x| Value::known(Fp::from(x)));
        let b = [2, 4, 6, 8, 10, 12, 14, 16].map(|x| Value::known(Fp::from(x)));

        let mux = Value::known(Fp::ONE);

        let circuit = MuxCircuit::<Fp, LEN> { a: &a, b: &b, mux };
        let k = 6;

        MockProver::run(k, &circuit, vec![])
            .unwrap()
            .assert_satisfied()
    }
}
